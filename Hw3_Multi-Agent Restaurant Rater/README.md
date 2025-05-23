# Multi-Agent Restaurant Rater

A system demonstrating a modular, multi-agent pipeline for rating restaurants based on review data. Originally built on Microsoft AutoGen with GPT-4o-mini, the pipeline has been optimized for direct invocation to ensure reliability and test compatibility.

## Table of Contents

- [Project Description](#project-description)  
- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Testing](#Testing)  
- [File-Structure](#file-structure)  


## Project Description

The Multi-Agent Restaurant Rater extracts numeric ratings from textual restaurant reviews by:

1. **Parsing** a natural-language query to identify the restaurant name.  
2. **Fetching** review lines from a structured dataset (`restaurant-data.txt`).  
3. **Analyzing** each review by mapping adjectives to food and service scores using keyword buckets.  
4. **Scoring** with a geometric-mean formula to compute an overall rating.

While the original design used conversational agents (Parse, Fetch, Analyze, Score), the current implementation calls helper functions directly in `main.py` to streamline execution and guarantee deterministic results.

## Features

- Robust name parsing for queries like:  
  - “How good is the restaurant Taco Bell overall?”  
  - “What is the overall score for In-n-Out?”  
- Keyword-based scoring with customizable adjective-to-score mappings.  
- Geometric-mean aggregation of food and service scores, rounded to three decimals.  
- Comprehensive test suite (`test.py`) for validating functionality.

## Prerequisites

- **Python** 3.8+  
- **OpenAI API Key** with access to `gpt-4o-mini`  
- **autogen** package version **0.9.0**

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/multi-agent-restaurant-rater.git
   cd multi-agent-restaurant-rater

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv           # create venv  
    source venv/bin/activate      # macOS/Linux  
    venv\Scripts\activate         # Windows  

3. Install dependencies:

    ```bash
    pip install -r requirements.txt

## Configuration

1. Get your OpenAI API key at https://platform.openai.com.

2. Set it as an environment variable:

    ```bash
    export OPENAI_API_KEY=sk-your-key   # macOS/Linux  
    set OPENAI_API_KEY=sk-proj-your-key      # Windows  

## Usage

1. Run `main.py`:

    ```bash
    python main.py restaurant-data.txt "How good is the restaurant Taco Bell overall?"


## Testing

1. Run the public tests to verify correctness:

    ```bash
    python test.py ./restaurant-data.txt

All tests should pass, ensuring parsing, scoring, and output formatting are correct.

## File Structure

```bash
├── main.py               # Core implementation  
├── test.py               # Public test suite  
├── restaurant-data.txt   # Sample review dataset  
├── requirements.txt      # Python dependencies  
└── README.md             # Project documentation  
