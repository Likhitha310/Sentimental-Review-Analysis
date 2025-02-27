Sentimental Review Analysis

This project implements a sentiment analysis system that utilizes machine learning and natural language processing techniques to evaluate and interpret the sentiment of textual data. The system is designed to analyze sentiments in various scenarios, including single-line reviews, multiple reviews from CSV files, and product reviews from Amazon.

Table of Contents
•	Features
•	Technologies Used
•	Project Structure
•	Setup Instructions
•	Usage
•	Contributing
•	License

Features
•	Versatile Input Handling: Capable of analyzing single-line reviews, multiple reviews from CSV files, and scraping and analyzing product reviews from Amazon.
•	Real-time Sentiment Analysis: Provides immediate feedback on the sentiment of the input text.
•	User-Friendly Interface: Offers an intuitive interface for users to interact with the system seamlessly.

Technologies Used
•	Programming Language: Python
•	Libraries: 
o	Scikit-learn: For building and training the machine learning model
o	Pandas: For data manipulation and analysis
o	NumPy: For numerical computations
o	BeautifulSoup & Requests: For web scraping
o	Streamlit: For creating the web application interface

Project Structure
The repository includes the following key files:
•	app.py: Main application script to run the Streamlit web app for sentiment analysis.
•	helper.py: Contains helper functions for data processing and analysis.
•	main.ipynb: Jupyter Notebook detailing the development and testing of the sentiment analysis model.
•	model.pkl: Serialized machine learning model for sentiment prediction.
•	tfidf.pkl: Serialized TF-IDF vectorizer used for text transformation.
•	req.txt: File listing all the dependencies required to run the project.

Setup Instructions
To set up the project on your local machine, follow these steps:
1.	Clone the Repository:
git clone https://github.com/Likhitha310/Sentimental-Review-Analysis.git
cd Sentimental-Review-Analysis
2.	Install Dependencies: Ensure you have Python installed. Install the required packages using:
pip install -r req.txt
3.	Run the Application: Start the Streamlit web application using:
streamlit run app.py
This will launch the web interface for sentiment analysis.

Usage
•	Single Review Analysis: Input a single line of text into the web app to receive an immediate sentiment analysis.
•	Batch Analysis: Upload a CSV file containing multiple reviews to analyze sentiments in bulk.
•	Amazon Product Reviews: Enter the URL of an Amazon product to scrape and analyze customer reviews.

Contributing
Contributions are welcome! If you'd like to enhance the project, please fork the repository, create a new branch, and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

