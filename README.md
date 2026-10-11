# Knowledge Desk

**Knowledge Desk** is a browser-based AI knowledge assistant that allows users to upload documents and ask questions about their content.

The application uses **client-side document retrieval with TF-IDF** and a **Hugging Face language model** to generate answers based only on the retrieved passages from the uploaded documents.

 ## Try Knowledge Desk Online
**Use the live application directly in your browser — no installation required.**
                  https://knowledge-desk.onrender.com

## Features

*  Upload PDF, TXT, and Markdown documents
*  Search documents using TF-IDF-based retrieval
*  Generate answers using a Hugging Face language model
*  Display the source documents used for each answer
*  Store documents and conversations in browser local storage
*  Drag-and-drop document uploading
*  Responsive interface for smaller screens
*  Hugging Face API token is stored locally in the browser
*  Retrieval-Augmented Generation (RAG) workflow
*  No dedicated backend server required

## Technologies Used

* **HTML5**
* **CSS3**
* **JavaScript**
* **PDF.js**
* **Hugging Face Inference API**
* **Llama 3.1 8B Instruct**
* **TF-IDF**
* **Cosine Similarity**
* **Browser LocalStorage**

## How It Works

Knowledge Desk follows a simple RAG pipeline:

```text
              User
                │
                ▼
       Upload Documents
                │
                ▼
       Text Extraction
                │
                ▼
        Text Chunking
                │
                ▼
       TF-IDF Indexing
                │
                ▼
        User Question
                │
                ▼
      Similarity Retrieval
                │
                ▼
       Relevant Passages
                │
                ▼
       Hugging Face LLM
                │
                ▼
        Grounded Answer
                │
                ▼
          Source Citations
```

## Supported Documents

The application currently supports:

* `.pdf`
* `.txt`
* `.md`

Uploaded documents are processed directly in the browser.

## How to Run

Download or open the project and launch `index.html` in a modern web browser.

For the best experience, use a local development server such as **VS Code Live Server**.

After opening the application:

1. Enter your Hugging Face API token.
2. Click **Save**.
3. Upload a PDF, TXT, or Markdown document.
4. Wait for the document to be indexed.
5. Enter a question related to the uploaded document.
6. Click **Ask**.
7. The application retrieves relevant passages and sends them to the selected Hugging Face model.
8. The generated answer and source passages are displayed in the interface.

## Hugging Face API Token

Knowledge Desk requires a Hugging Face API token for answer generation.

The token is entered through the application's interface and stored in the browser's `localStorage`.

**Important:** Never hard-code your API token into `index.html` before pushing the project to GitHub.

Do not commit:

```text
hf_xxxxxxxxxxxxxxxxx
```

or any other secret API credentials.

## Retrieval System

The application uses a lightweight client-side retrieval system.

Documents are:

1. Extracted
2. Cleaned
3. Divided into overlapping chunks
4. Tokenized
5. Indexed using TF-IDF
6. Compared against the user's question using cosine similarity

The most relevant passages are then provided to the language model as context.

## AI Model

The current implementation uses:

```text
meta-llama/Llama-3.1-8B-Instruct
```

through the Hugging Face Inference Providers router.

The model is instructed to answer using only the retrieved document context and avoid guessing when the required information is not available.

## Data Storage

Knowledge Desk uses browser `localStorage` to save:

* Uploaded document indexes
* Conversation history
* Hugging Face API token

This means the data remains available when the application is opened again in the same browser.

## Privacy

Documents are processed in the browser for text extraction and retrieval.

However, relevant retrieved passages are sent to the Hugging Face API when an answer is generated.

Therefore, do not upload confidential or sensitive documents unless you understand the data-sharing implications of the model/API you are using.

## Project Structure

```text
Knowledge-Desk/
│
├── index.html       # Main application
├── README.md        # Project documentation
├── .gitignore       # Files excluded from Git
└── LICENSE          # Project license
```

## Future Improvements

Possible future improvements include:

* Semantic embeddings instead of TF-IDF
* Vector database integration
* Better document parsing
* Multiple AI model selection
* Streaming AI responses
* Authentication
* Backend API
* Cloud document storage
* Conversation export
* Markdown-formatted AI responses
* Better PDF support
* OCR for scanned PDFs
* Deployment using GitHub Pages

## Learning Objectives

This project demonstrates practical concepts in:

* Artificial Intelligence
* Large Language Models
* Retrieval-Augmented Generation
* Natural Language Processing
* Information Retrieval
* TF-IDF
* Cosine Similarity
* API Integration
* Frontend Development
* Browser Storage
* Document Processing

## Author

**Praveen Kumar kunwar**

GitHub: `praveenkumarkunwar2007-hash`



