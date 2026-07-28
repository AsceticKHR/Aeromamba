import urllib.request

def peek(path, nbytes=3500):
    req = urllib.request.Request(
        "https://huggingface.co/datasets/" + path,
        headers={"Range": "bytes=0-%d" % nbytes},
    )
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")

print("=== airspatial rec_train ===")
for line in peek("erenzhou/AirSpatial/resolve/main/airspatial_rec_train.jsonl").splitlines()[:3]:
    print(line[:800]); print("---")

print("=== hrvqa train_question ===")
print(peek("JNIC1/HRVQA/resolve/main/jsons/train_question.json", 800)[:800])
print("=== hrvqa train_answer ===")
print(peek("JNIC1/HRVQA/resolve/main/jsons/train_answer.json", 800)[:800])
