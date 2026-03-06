# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-12-26

import os
import json
import http.client
import urllib.parse
from typing import List, Dict
from dotenv import load_dotenv

class HTBEnv:
    def __init__(self):
        self.conn = http.client.HTTPSConnection("labs.hackthebox.com")
        self.token = os.getenv("HTB_TOKEN")
        
    # Challenges
    def get_challenge_list(self) -> List[Dict]:
        payload = ''
        headers = {'Authorization': f'Bearer {self.token}'}
        self.conn.request("GET", "/api/v4/challenge/list", payload, headers)
        res = self.conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))
    
    def get_challenge_info(self, challenge_id: str) -> Dict:
        payload = ''
        headers = {'Authorization': f'Bearer {self.token}'}
        self.conn.request("GET", f"/api/v4/challenge/info/{challenge_id}", payload, headers)
        res = self.conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))
    
    def get_challenge_file(self, challenge_id: str):
        payload = ''
        headers = {'Authorization': f'Bearer {self.token}'}
        self.conn.request("GET", f"/api/v4/challenge/download/{challenge_id}", payload, headers)
        res = self.conn.getresponse()
        data = res.read()
        filename = f"htb_challenge_{challenge_id}.zip"
        with open(filename, "wb") as f:
            f.write(data)
        return filename
    
    def post_challenge_start(self, challenge_id: str) -> Dict:
        payload = json.dumps({"challenge_id": challenge_id})
        headers = {'Authorization': f'Bearer {self.token}', 'Content-Type': 'application/json'}
        self.conn.request("POST", f"/api/v4/challenge/start", payload, headers)
        res = self.conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))
    
    def post_challenge_flag(self, challenge_id: str, flag: str, difficulty: str) -> Dict:
        payload = json.dumps({
            "challenge_id": challenge_id,
            "flag": flag,
            "difficulty": difficulty
        })
        headers = {'Authorization': f'Bearer {self.token}', 'Content-Type': 'application/json'}
        self.conn.request("POST", f"/api/v4/challenge/own", payload, headers)
        res = self.conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))

    
if __name__ == "__main__":
    load_dotenv('/Users/mingzhe/Projects/mastermind/.env')
    htb = HTBEnv()
    # print(htb.get_challenge_list())
    # print(htb.get_challenge_info("1042"))
    # print(htb.post_challenge_start("1042"))
    # print(htb.get_challenge_file("1042"))
    print(htb.post_challenge_flag("1042", "HTB{token}", "10"))