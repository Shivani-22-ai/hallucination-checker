# LLM Hallucination Checker

An NLP-based system for checking whether an LLM-generated answer is supported by a given source document.

## Overview

Large Language Models can generate answers that sound convincing but are not supported by the provided information.

This project analyzes generated answers at the claim level and checks each claim against relevant evidence from the source document.

## How It Works

```text
Source Document
       ↓
Source Chunking
       ↓
LLM Generated Answer
       ↓
Claim Extraction
       ↓
Semantic Retrieval
       ↓
Relevant Evidence
       ↓
NLI Classification
       ↓
Supported / Contradicted / Unsupported
       ↓
Factual Consistency Score