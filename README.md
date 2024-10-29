-Add the Key to Hugging Face:
	•	Go to Hugging Face SSH Key Settings.
	•	In the SSH Keys section, click Add a new SSH key.
	•	Paste the content of your huggingfacelllm.pub key into the provided text field.
	•	Give the key a descriptive name, e.g., huggingfacelllm, and click Save.
	3.	Ensure Your SSH Key is Loaded:
-Ensure the private key (huggingfacelllm) is loaded into your SSH agent:


  > ssh-add ~/.ssh/huggingfacelllm
  > ssh -T git@hf.co
    [Hi <your username>! You've successfully authenticated, but Hugging Face does not provide shell access.
    
    ]
> git clone git@hf.co:meta-llama/Llama-3.2-3B 

-Continue working with the existing directory

If you already have the repository partially cloned and just want to pull any missing LFS files, you can navigate to the directory and use the git lfs pull command to download the large files managed by Git LFS:

>cd Llama-3.2-3B
>git lfs pull


This uses FAISS Vector DB store
