# Web Scrapping with Beautiful Soup

## Introduction

Web scraping is the process of extracting data from websites. The purpose of web scraping is to fetch data from any website thereby saving a huge amount of manual labour in collecting data/information. Here we are using Beautiful Soup for web scrapping. In this project the scrapped data is saved in a json file.

## Requirements

Install required libraries using pip:

```
pip install beautifulsoup4
pip install lxml
pip install requests

```

## General Usage

1.  **Import required libraries and functions:**

    ```python
    from bs4 import BeautifulSoup
    import requests
    import json
    import os

    ```

2.  **Sending an HTTP Request**
    To scrape data from a website, we need to send an HTTP GET request to the URL of the webpage we want to scrape. This request retrieves the HTML content of the page.

    ```python
    url = 'https://example.com'
        response = requests.get(url)
        html_text = response.text
    ```

3.  **Parsing HTML Content**
    Once we have the HTML content, we can parse it using BeautifulSoup. BeautifulSoup provides methods to navigate and extract data from the HTML document easily.

    ```python
    soup = BeautifulSoup(html_text, 'lxml')
    ```

4.  **Locating Data**
    To locate the specific data we want to scrape within the HTML document involves identifying HTML elements, their attributes, and their hierarchy.

    ```python
    data = soup.find('div', class_='example-class')

    ```

5.  **Extracting Data**
    Once we've located the data, we can extract it using BeautifulSoup methods. For example, we can extract text or attribute values.

    ```python
    text = data.text
    attribute_value = data['attribute_name']

    ```

6.  **Storing Data**
    After extracting the data, we can store it in an appropriate data structure, such as lists or dictionaries, for further processing or saving.

    ```python
    data_list = []
    data_list.append(data)
    ```

7.  **Handling Pagination (if necessary)**
    Finally, we can save the scraped data to a file format of your choice, such as JSON or CSV, for future analysis or use.
    ```python
    with open('data.json', 'w') as json_file:
    json.dump(data_list, json_file)
    ```

## References

- [Beautiful Soup Crash Course from freecodecamp](https://www.youtube.com/watch?v=XVv6mJpFOb0&ab_channel=freeCodeCamp.org)
- [Web Scraping with BeautifulSoup and Requests from Corey Schafer](https://www.youtube.com/watch?v=ng2o98k983k&ab_channel=CoreySchafer)
- [Web Scraping a Site with Pagination using BeautifulSoup
  ](https://medium.com/analytics-vidhya/webscraping-a-site-with-pagination-using-beautifulsoup-fa0a09804445)
