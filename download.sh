# Download the datasets and checkpoints

if [ ! -d datasets/ycb/YCB_Video_Dataset ];then
echo 'Downloading the YCB-Video Dataset'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi" -O YCB_Video_Dataset.zip && rm -rf /tmp/cookies.txt \
&& unzip YCB_Video_Dataset.zip \
&& mv YCB_Video_Dataset/ datasets/ycb/ \
&& rm YCB_Video_Dataset.zip
fi

if [ ! -d datasets/linemod/Linemod_preprocessed ];then
echo 'Downloading the preprocessed LineMOD dataset'
wget --load-cookies /tmp/cookies.txt "https://drive.usercontent.google.com/download?id=1YFUra533pxS_IHsb9tB87lLoxbcHYXt8&export=download&authuser=0&confirm=t&uuid=61b761c6-02b7-427c-b66c-b7731f61ee27&at=APvzH3qO_R1HBNljzJT1iAAXKKG9:1735995013608" -O Linemod_preprocessed.zip && rm -rf /tmp/cookies.txt \
&& unzip Linemod_preprocessed.zip \
&& mv Linemod_preprocessed/ datasets/linemod/ \
&& rm Linemod_preprocessed.zip
fi

if [ ! -d trained_checkpoints ];then
echo 'Downloading the trained checkpoints...'
wget --load-cookies /tmp/cookies.txt "https://drive.usercontent.google.com/download?id=1gfOnOojzVdEwPzSaPmS3t3aJaQptbys6&export=download&authuser=0&confirm=t&uuid=fc130b6f-34d4-474e-a56a-d6a4e1f9e439&at=APvzH3rfATC3WyxXvO6HLgkXrIER:1735993491936" -O trained_checkpoints.zip && rm -rf /tmp/cookies.txt \
&& unzip trained_checkpoints.zip -x "__MACOSX/*" "*.DS_Store" "*.gitignore" -d trained_checkpoints \
&& mv trained_checkpoints/trained*/ycb trained_checkpoints/ycb \
&& mv trained_checkpoints/trained*/linemod trained_checkpoints/linemod \
&& rm -r trained_checkpoints/trained*/ \
&& rm trained_checkpoints.zip
fi

echo 'done'