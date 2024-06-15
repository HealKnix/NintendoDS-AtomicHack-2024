from fastapi import FastAPI
import hashlib

app = FastAPI()

@app.get("/")
def root(key: str):
  hash = hashlib.sha256()
  hash.update(key.encode())
  
  if hash.hexdigest() == "74323a1db24ecc5a514efa9b8409e0f899249d76cbfba51ce252595818d8b2c3":
    print('yes')
  else:
    print('no')
  
  return {"Hello": "World"}
