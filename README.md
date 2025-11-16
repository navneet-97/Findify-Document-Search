# Findify - Smart Document Search

Findify is a smart document search tool that helps you organize and find your documents quickly. It can search across multiple file formats and automatically categorizes documents by topic.

## Features

- **Smart Search**: Search documents by keywords, file names, or content
- **Multiple Formats**: Works with PDFs, Word docs, PowerPoint, text files, and images
- **Auto Categorization**: Automatically tags documents by topic (marketing, sales, product, etc.)
- **File Type Filtering**: Filter documents by type (PDF, Word, images, etc.)
- **Tag-Based Organization**: Documents are automatically tagged for easy filtering
- **Instant Preview**: See document snippets in search results
- **Upload & Download**: Easily upload new documents and download existing ones

## How to Use

1. **Upload Documents**: Click the "Upload" button to add documents to your library
2. **Search**: Type in the search bar to find documents by keywords or phrases
3. **Filter**: Use the filter button to narrow down results by file type or tags
4. **Download**: Click "Download" on any document to save it to your computer
5. **Delete**: Remove documents you no longer need with the delete button

## Supported File Types

- PDF documents
- Microsoft Word (.docx)
- Microsoft PowerPoint (.pptx)
- Text files (.txt)
- Images (.png, .jpg, .jpeg)

## Technology Stack

- **Frontend**: React with TailwindCSS
- **Backend**: Python FastAPI
- **Database**: MongoDB for document metadata
- **Search**: ChromaDB for semantic search
- **AI**: Sentence Transformers for document embeddings

## Setup Instructions

### Backend Setup

1. Install Python 3.9+
2. Install dependencies: `pip install -r requirements.txt`
3. Set up MongoDB and update the connection string in `.env`
4. Run the server: `uvicorn server:app --host 0.0.0.0 --port 8000`

### Frontend Setup

1. Install Node.js
2. Install dependencies: `npm install`
3. Update the backend URL in `.env`
4. Start the app: `npm start`

## Deployment

- **Backend**: Can be deployed on Render
- **Frontend**: Can be deployed on Vercel

## Need Help?

If you have any questions or run into issues, check the documentation or contact support.