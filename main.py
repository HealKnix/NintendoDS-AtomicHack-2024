from fastapi import FastAPI
import hashlib
import re

app = FastAPI()

@app.get("/")
async def root(key: str, text: str):
  hash = hashlib.sha256()
  hash.update(key.encode())
  
  file_name_not_formatted = "(Источник: docs/Отражение операций по давальческой схеме через документ «Заказ давальца»_v2.pdf)"
  
  reg = re.search(r'(?=docs/).*?(?=\.pdf)', file_name_not_formatted)
  
  file_name_formatted = reg[0].replace('docs/', '')
  
  if hash.hexdigest() == "74323a1db24ecc5a514efa9b8409e0f899249d76cbfba51ce252595818d8b2c3":
    return {
      "access": "accept",
      "text_req": text,
      "text_res": "Извините, я пока что молчун =(",
      "file_name": file_name_formatted
    }
  else:
    return {
      "access": "deny",
      "text_req": "Ага, размечтался",
      "text_res": "Это неправильный код",
      "file_name": ":p"
    }
