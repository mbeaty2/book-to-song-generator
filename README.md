# Developing a Book-to-Song Content Based Generator with Spotify's API

This project seeks to develop a Book-to-Song content-based generator using Spotify's API and Natural Language Processing.

## Description

The primary objective of this project was to construct a recommendation system capable of suggesting songs based on the content of a book. To accomplish this, I utilized a Kaggle dataset that contained the text and genres of the chosen book, Jane Austen's "Persuasion," and a custom playlist on Spotify featuring 78 songs representing various major genre groupings. The project's foundation was laid during the "Personalization and Machine Learning Week 3.1" course. Due to the project's time constraints, I opted to concentrate on generating song recommendations for a single selected book. To bridge the gap between the textual content of the book and song features, I carefully selected four specific features: Energy, Valence, Instrumentalness, and Speechiness. The initial focus was on defining Speechiness and Instrumentalness. Spotify defines Speechiness as the frequency of words in a song, which I equated with wordiness in text. I explored two interpretations: one based on the ratio of unique words to the total word count and another based on the ratio of stopwords to total words. The latter proved to be a more accurate representation. Instrumentalness, on the other hand, presented a challenge since books inherently contain "vocals." To address this, I defined it as a ratio of the book's total word count relative to the longest book (in terms of word count). Subsequently, I modified this definition to consider the book's word count relative to the average word count of books across various genres. For Valence, the sentiment analysis tool Flair was employed, resulting in a sentiment value. Energy, the most complex feature to define, was approached by calculating the cosine similarity between the book's genre descriptions and the term "energy." This method involved data cleaning and resulted in an energy value.

Once these values were established, I compared the book's features with those derived from songs in my playlist. I created four book variations, accounting for different interpretations of Speechiness and Instrumentalness. When examining the song recommendations for these variations, the results appeared to correlate with the values associated with each book. This indicated that the recommendation system itself was not flawed, but rather the features derived for the book were perhaps subjective. The alternative perspective suggested that the subjective association between books and songs is influenced by preconceived notions and emotional responses to the book, which cannot be entirely captured by quantitative features.

To fit the hand-in specifications of my course which this project was developed for, the project was originally written and tested within a jupyter notebook that has been uploaded to this repository. This notebook has also been adapted to python files to run in the command line. A full report of the project findings has also been attached as a PDF.

## Getting Started

### Dependencies

*  Must acquire a Spotify API to run the code: https://developer.spotify.com/documentation/web-api

### Installing

* A full list of dependencies is available in the python file and jupyter notebook, though the project requires the installation of:
     * matplotlib, pandas, numpy, nltk, and flair
     * among others.
* The book dataset is available here: https://www.kaggle.com/datasets/meetnaren/goodreads-best-books
* The spotify song-list is available here: https://open.spotify.com/playlist/4eIxXnvi5KeTMPAzcgwSud?si=d057c60b2e6e4c4d

### Executing program

* Download the necessary datasets. 

If using the Jupyter Notebook:
* Open the file in a code editor and connect to a code environment.
* Run each cell in the notebook. Should you want to make adjustments, be sure to save the file before continuing to run each cell. If there are issues, restart the kernel and re-run each cell.

If using the python files:
* Download the .py files
* Open a terminal and locate where the downloaded datasets and files are on your local machine.
* Create a new environment to run everything in.
* Run this line in your terminal: ADD

## Help

While Spotify utilizes multiple song features for its recommendations, some features cannot be seamlessly translated to books. In future iterations of this project, I would explore additional features and more intricate definitions, especially for "Energy," to potentially yield different and more accurate results. Nevertheless, the current system successfully achieved the goal of recommending songs based on a book. Despite the potential for fine-tuning and refinement, this project demonstrated the feasibility of establishing connections between books and songs. The results, while not flawless, are both intriguing and thought-provoking, ultimately marking this project as a success.

## Authors

Marissa Beaty, https://github.com/mbeaty2

## Version History

* 0.2
    * translation of the project from a jupyter notebook to a python file for mass use and adaptation.
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
