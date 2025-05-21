# Chroma Photo Search

I used this to create an AI assistant that I can ask about my Dropbox photo collection.

It allows you to create a Chroma vector database from a folder of images, then use it for RAG with an LLM.

This is a small demo project simply used to learn about Chroma, vector databases and RAG.

## Usage

### Create Vector DB

```bash
uv run main.py create --folder "/path/to/folder/of/images"
```

### Query using RAG

```bash
uv run main.py query --query "Have I ever been to London?"
```
