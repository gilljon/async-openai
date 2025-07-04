//! Rust library for OpenAI
//!
//! ## Creating client
//!
//! ```
//! use async_openai::{Client, config::OpenAIConfig};
//!
//! // Create a OpenAI client with api key from env var OPENAI_API_KEY and default base url.
//! let client = Client::new();
//!
//! // Above is shortcut for
//! let config = OpenAIConfig::default();
//! let client = Client::with_config(config);
//!
//! // OR use API key from different source and a non default organization
//! let api_key = "sk-..."; // This secret could be from a file, or environment variable.
//! let config = OpenAIConfig::new()
//!     .with_api_key(api_key)
//!     .with_org_id("the-continental");
//!
//! let client = Client::with_config(config);
//!
//! // Use custom reqwest client
//! let http_client = reqwest::ClientBuilder::new().user_agent("async-openai").build().unwrap();
//! let client = Client::new().with_http_client(http_client);
//! ```
//!
//!
//! ## Making requests
//!
//!```
//!# tokio_test::block_on(async {
//!
//! use async_openai::{Client, types::{CreateCompletionRequestArgs}};
//!
//! // Create client
//! let client = Client::new();
//!
//! // Create request using builder pattern
//! // Every request struct has companion builder struct with same name + Args suffix
//! let request = CreateCompletionRequestArgs::default()
//!     .model("gpt-3.5-turbo-instruct")
//!     .prompt("Tell me the recipe of alfredo pasta")
//!     .max_tokens(40_u32)
//!     .build()
//!     .unwrap();
//!
//! // Call API
//! let response = client
//!     .completions()      // Get the API "group" (completions, images, etc.) from the client
//!     .create(request)    // Make the API call in that "group"
//!     .await
//!     .unwrap();
//!
//! println!("{}", response.choices.first().unwrap().text);
//! # });
//!```
//!
//! ## Bring Your Own Types
//!
//! To use custom types for inputs and outputs, enable `byot` feature which provides additional generic methods with same name and `_byot` suffix.
//! This feature is available on methods whose return type is not `Bytes`
//!
//!```
//!# #[cfg(feature = "byot")]
//!# tokio_test::block_on(async {
//! use async_openai::Client;
//! use serde_json::{Value, json};
//!
//! let client = Client::new();
//!
//! let response: Value = client
//!        .chat()
//!        .create_byot(json!({
//!            "messages": [
//!                {
//!                    "role": "developer",
//!                    "content": "You are a helpful assistant"
//!                },
//!                {
//!                    "role": "user",
//!                    "content": "What do you think about life?"
//!                }
//!            ],
//!            "model": "gpt-4o",
//!            "store": false
//!        }))
//!        .await
//!        .unwrap();
//!
//!  if let Some(content) = response["choices"][0]["message"]["content"].as_str() {
//!     println!("{}", content);
//!  }
//! # });
//!```
//!
//! ## Microsoft Azure
//!
//! ```
//! use async_openai::{Client, config::AzureConfig};
//!
//! let config = AzureConfig::new()
//!     .with_api_base("https://my-resource-name.openai.azure.com")
//!     .with_api_version("2023-03-15-preview")
//!     .with_deployment_id("deployment-id")
//!     .with_api_key("...");
//!
//! let client = Client::with_config(config);
//!
//! // Note that `async-openai` only implements OpenAI spec
//! // and doesn't maintain parity with the spec of Azure OpenAI service.
//!
//! ```
//!
//!
//! ## Examples
//! For full working examples for all supported features see [examples](https://github.com/64bit/async-openai/tree/main/examples) directory in the repository.
//!
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "byot")]
pub(crate) use async_openai_macros::byot;

#[cfg(not(feature = "byot"))]
pub(crate) use async_openai_macros::byot_passthrough as byot;

mod assistants;
mod audio;
mod audit_logs;
mod batches;
mod chat;
mod client;
mod completion;
pub mod config;
mod download;
mod embedding;
pub mod error;
mod file;
mod fine_tuning;
mod image;
mod invites;
mod messages;
mod model;
mod moderation;
mod project_api_keys;
mod project_service_accounts;
mod project_users;
mod projects;
mod runs;
mod steps;
mod threads;
mod tokenize;
pub mod traits;
pub mod types;
mod uploads;
mod users;
mod util;
mod vector_store_file_batches;
mod vector_store_files;
mod vector_stores;

pub use assistants::Assistants;
pub use audio::Audio;
pub use audit_logs::AuditLogs;
pub use batches::Batches;
pub use chat::Chat;
pub use client::Client;
pub use completion::Completions;
pub use embedding::Embeddings;
pub use file::Files;
pub use fine_tuning::FineTuning;
pub use image::Images;
pub use invites::Invites;
pub use messages::Messages;
pub use model::Models;
pub use moderation::Moderations;
pub use project_api_keys::ProjectAPIKeys;
pub use project_service_accounts::ProjectServiceAccounts;
pub use project_users::ProjectUsers;
pub use projects::Projects;
pub use runs::Runs;
pub use steps::Steps;
pub use threads::Threads;
pub use tokenize::Tokenize;
pub use uploads::Uploads;
pub use users::Users;
pub use vector_store_file_batches::VectorStoreFileBatches;
pub use vector_store_files::VectorStoreFiles;
pub use vector_stores::VectorStores;
