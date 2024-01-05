# CloudVault

## Elevating Banking Security with End-to-End Cloud Solutions

We offer a cloud native platform for financial institutions to securely launch Credit Card Programs, Perform Fraud Detection and Underwriting for sanctioning loans.

Here's the deployed application: https://encryptcon.vercel.app.

Here's the Cloud Native REST APIs: https://api-encryptcon.jayanthk.in.

The documentation of the APIs can be viewed here: https://api-encryptcon.jayanthk.in/docs.

This is the detailed article about our project: https://substantial-nymphea-a72.notion.site/Final-Submission-efce2f425966462c9975ca5295e105db?pvs=4.

### Tech Stack

```
- Next JS, Tailwind CSS & ShadCN UI
- Python, FastAPI, Pydantic
- Pytorch, DLG, Pandas, Numpy, Sklearn
- AWS Cloud
```

### Installation Instructions

- Clone the repository.
- To start the client, run
  ```
  cd client
  yarn install
  yarn dev
  ```
- To start the server, run
  ```
  cd server
  python -m venv venv
  pip install -r requirements.txt
  uvicorn main:app --reload
  ```
- There is a simulation for transactions. To run it, run
  ```cd transaction-api
    python -m venv api_env
    pip install -r requirements.txt
    uvicorn main:app --reload -- port 8001
  ```
