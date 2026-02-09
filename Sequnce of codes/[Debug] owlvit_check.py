from transformers import OwlViTProcessor
print("High-level methods:")
for m in dir(OwlViTProcessor):
    if "post_process" in m:
        print(f" - {m}")
