# Voice of Customer Marketing Assistant

This demonstrates using Astra as a vector store to support a marketing person. Using real-world customer review data (scraped from Amazon in 2018), 
the demo shows how marketing material can be used to provide messaging in the "[voice of the customer](https://www.qualtrics.com/uk/experience-management/customer/voice-of-customer/)".

In contains both a Streamlit-based user interface, as well as a Jupyter notebook that can be used to explore the data and the model.

## Python and Jupyter Environment
Determine your Python envrionment. It is suggested to use venv or Conda; the example code was developed with Python 3.11.4. Similarly,
you should have an environment with Jupyter installed.

## Open the Setup Notebook
Open [setup.ipynb](setup.ipynb) in a Jupyter-compatible viewer, and follow the instructions there.

### Other Scripts
There are a few other notebooks within this repository that are not used by the setup or demo, but were used to create the dataset 
and embeddings.

* [`cutdownReviews.ipynb`](cutdownReviews.ipynb) - Referencing JSON-formatted reviews, this notebook documents how the the reviews
    were cut down to a more manageable size. It also removes reviews that do not have any text (i.e. they are a rating only),
    and creates a `truncated` version of the review that is limited to 400 characters (and is used for embeddings). The reviews
    and associated metadta are saved in `.parquet.gz` files which available by downloading the `.zip` file described in the setup notebook.
* [`createEmbeddings.ipynb`](createEmbeddings.ipynb) - Referencing the cutdown reviews, this notebook selects a single product and 
    generates OpenAI embeddings of the truncated review text. This is saved in an embeddgings `.parquet.gz` file.

## Run the Demos
### Jupyter Notebook
Once the setup is complete, open [demo.ipynb](demo.ipynb) in a Jupyter-compatible viewer and follow the instructions there. The notebook 
gives a step-by-step "under the covers" description of what is happening.

### Streamlit UI
Alternately, a simple UI allows you to interact with the demo in a more presentation-friendly manner. To run the UI, use the following command:

```
streamlit run demo-ui.py
```

## Credits
This demo is based on a [YouTube video](https://www.youtube.com/watch?v=fCh7PKR5WqU) from [@rabbitmetrics](https://www.youtube.com/@rabbitmetrics). 

It in turn has used a customer review dataset scraped from Amazon in 2018, documented [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) and compiled as part of the following paper:

**Justifying recommendations using distantly-labeled reviews and fined-grained aspects**\
Jianmo Ni, Jiacheng Li, Julian McAuley\
_Empirical Methods in Natural Language Processing (EMNLP), 2019_\
[pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf)