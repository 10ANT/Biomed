echo "Current directory contents:"
ls -l
echo "Contents of target_dist.json:"
cat target_dist.json #This will print the contents of the json file to the docker logs.

#!/bin/bash
if [ -f "/run/secrets/HF_TOKEN" ]; then
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN)
fi
exec conda run --no-capture-output -n biomedparse python main.py
