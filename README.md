# Messenger Text Extraction and Clustering

## Overview
This project aims to develop a Python program that extracts text from messenger screenshots, preferably from Telegram, utilizing the Yandex Vision OCR API. The extracted text is then processed using the DBSCAN algorithm to segment the image into clusters. These clusters help classify user messages and filter out extraneous text, such as timestamps and other non-user messages, from the screenshot.

## Examples
![Image alt](https://github.com/Thunder17/messenger-screenshot-parser/raw/main/pasesr_sample.jpg)

## Features
- Extracts text from messenger screenshots using Yandex Vision OCR API.
- Utilizes the DBSCAN algorithm for pixel-wise clustering of the image.
- Classifies user messages and filters out non-user text elements.
- Supports Telegram messenger screenshots as primary input.
- Provides clear classification and filtering of messages for enhanced readability.

## Installation
1. Clone the repository:
   git clone https://github.com/Thunder17/messenger-screenshot-parser
2. Navigate to the project directory:
   cd messenger-screenshot-parser
3. Install the required dependencies:
   pip install -r requirements.txt


## Usage
1. Acquire a screenshot from a messenger application, preferably Telegram.
2. Run the program and provide the screenshot as input as path to save the result.
    python main.py path/to/screenshot.png path/to/result
