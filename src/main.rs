use clap::{Parser, Subcommand};
use crossbeam_channel::bounded;
use futures::executor::block_on;
use std::fs;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Write};
use std::process;
use std::thread;
use std::time::Instant;
use tantivy::merge_policy::NoMergePolicy;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field, FieldType, IndexRecordOption, SchemaBuilder, TextFieldIndexing, TextOptions,
};
use tantivy::{Index, TERMINATED};
use tracing::{error, info};

const INDEX_DIR: &'static str = "data/index";

#[derive(Debug, Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Create {
        #[clap(long)]
        overwrite: bool,
    },
    Index {
        #[clap(long, default_value_t = 4)]
        num_threads: usize,
    },
    Merge {
        #[clap(long, default_value_t = 1)]
        num_segments: usize,
    },
    #[clap(visible_alias = "gc")]
    GarbageCollect,
    Search {
        #[clap(short, long)]
        query: String,
    },
}

fn create(overwrite: bool) -> anyhow::Result<()> {
    if overwrite {
        fs::remove_dir_all(INDEX_DIR)?;
    }
    fs::create_dir(INDEX_DIR)?;

    let mut schema_builder = SchemaBuilder::default();

    let url_indexing_options = TextFieldIndexing::default()
        .set_index_option(IndexRecordOption::Basic)
        .set_tokenizer("raw");
    let url_options = TextOptions::default()
        .set_indexing_options(url_indexing_options)
        .set_stored();

    let title_indexing_options = TextFieldIndexing::default()
        .set_index_option(IndexRecordOption::WithFreqsAndPositions)
        .set_tokenizer("raw");
    let title_options = TextOptions::default()
        .set_indexing_options(title_indexing_options)
        .set_stored();

    let body_indexing_options = TextFieldIndexing::default()
        .set_index_option(IndexRecordOption::WithFreqsAndPositions)
        .set_tokenizer("en_stem");
    let body_options = TextOptions::default()
        .set_indexing_options(body_indexing_options)
        .set_stored();

    schema_builder.add_text_field("url", url_options);
    schema_builder.add_text_field("title", title_options);
    schema_builder.add_text_field("body", body_options);

    let schema = schema_builder.build();
    Index::create_in_dir(INDEX_DIR, schema)?;
    Ok(())
}

fn index(num_threads: usize) -> anyhow::Result<()> {
    let index = Index::open_in_dir(INDEX_DIR)?;
    let schema = index.schema();
    let (parser_tx, parser_rx) = bounded(10_000);
    let (indexer_tx, indexer_rx) = bounded(10_000);

    // Read lines from stdin in one dedicated thread.
    info!("Spawning reader thread.");
    thread::spawn(move || {
        let stdin = io::stdin();
        let reader = BufReader::new(stdin);

        for line in reader.lines() {
            parser_tx.send(line.unwrap()).unwrap();
        }
    });

    // Parse lines to JSON docs in `parser_num_threads` dedicated threads.
    let parser_num_threads = 1.max(num_threads / 4);
    info!(
        num_threads = parser_num_threads,
        "Spawning parser thread(s)."
    );

    for _ in 0..parser_num_threads {
        let schema = schema.clone();
        let parser_rx = parser_rx.clone();
        let indexer_tx = indexer_tx.clone();

        thread::spawn(move || {
            for line in parser_rx {
                match schema.parse_document(&line) {
                    Ok(doc) => {
                        indexer_tx.send((line.len(), doc)).unwrap();
                    }
                    Err(err) => {
                        error!(
                            "Failed to parse document `{}...`: {:?}",
                            line.chars().take(20).collect::<String>(),
                            err
                        );
                    }
                }
            }
        });
    }
    drop(indexer_tx);

    let indexer_num_threads = 1.max(num_threads);
    let mut index_writer =
        index.writer_with_num_threads(indexer_num_threads, 1 * 1024 * 1024 * 1024)?;
    index_writer.set_merge_policy(Box::new(NoMergePolicy));

    let start = Instant::now();
    let mut num_bytes = 0;
    let mut num_docs = 0;

    for (doc_len, doc) in indexer_rx {
        index_writer.add_document(doc)?;
        num_bytes += doc_len;
        num_docs += 1;

        if num_docs % 100_000 == 0 {
            println!(
                "{} docs. Throughput: {:.2} MiB/s,  {:.0} docs/hour.",
                num_docs,
                num_bytes as f64 / start.elapsed().as_secs_f64() / 1024.0 / 1024.0,
                num_docs as f64 / start.elapsed().as_secs_f64() * 3600.0,
            );
        }
    }
    let opstamp = index_writer.commit()?;
    info!(
        num_bytes = num_bytes,
        num_docs = num_docs,
        num_secs = start.elapsed().as_secs(),
        opstamp = opstamp,
        "Indexing complete."
    );
    Ok(())
}

fn merge(_num_segments: usize) -> anyhow::Result<()> {
    let index = Index::open_in_dir(INDEX_DIR)?;
    let segment_ids = index.searchable_segment_ids()?;
    let mut index_writer = index.writer(8 * 1024 * 1024 * 1024)?;
    block_on(index_writer.merge(&segment_ids))?;
    Ok(())
}

fn gc() -> anyhow::Result<()> {
    let index = Index::open_in_dir(INDEX_DIR)?;
    let index_writer = index.writer_with_num_threads(1, 30 * 1024 * 1024)?;
    block_on(index_writer.garbage_collect_files())?;
    Ok(())
}

fn search(query: &str) -> anyhow::Result<()> {
    let index = Index::open_in_dir(INDEX_DIR)?;
    let schema = index.schema();
    let default_fields: Vec<Field> = schema
        .fields()
        .filter(|&(_, ref field_entry)| match *field_entry.field_type() {
            FieldType::Str(ref text_field_options) => {
                text_field_options.get_indexing_options().is_some()
            }
            _ => false,
        })
        .map(|(field, _)| field)
        .collect();
    let query_parser = QueryParser::new(schema.clone(), default_fields, index.tokenizers().clone());
    let query = query_parser.parse_query(query)?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(&searcher, true)?;

    let mut stdout = BufWriter::new(io::stdout());

    for segment_reader in searcher.segment_readers() {
        let mut scorer = weight.scorer(segment_reader, 1.0)?;
        let store_reader = segment_reader.get_store_reader()?;

        while scorer.doc() != TERMINATED {
            let doc_id = scorer.doc();
            let doc = store_reader.get(doc_id)?;
            let named_doc = schema.to_named_doc(&doc);

            if let Err(error) = writeln!(stdout, "{}", serde_json::to_string(&named_doc).unwrap()) {
                if error.kind() != ErrorKind::BrokenPipe {
                    eprintln!("{}", error);
                    process::exit(1)
                }
            }
            scorer.advance();
        }
    }
    if let Err(error) = stdout.flush() {
        if error.kind() != ErrorKind::BrokenPipe {
            eprintln!("{}", error);
            process::exit(1)
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Create { overwrite } => create(overwrite)?,
        Commands::Index { num_threads } => index(num_threads)?,
        Commands::Merge { num_segments } => merge(num_segments)?,
        Commands::GarbageCollect => gc()?,
        Commands::Search { query } => search(&query)?,
    }
    Ok(())
}
