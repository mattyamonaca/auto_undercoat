# auto_undercoat
"auto_undercoat" is an automatic undercoat and layer splitter

auto_undercoatは、入力された線画に対してフラットな色付け（下塗り）を行うツールです。
また、下塗りの色情報に基づいて、自動でレイヤー分けを行い透過PNGで出力することが可能です。

![スクリーンショット 2023-06-18 083200](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/b87f9a90-ca81-4947-a558-9bc7fac5071c)

![スクリーンショット 2023-06-18 083356](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/f9745298-e428-4ff7-a3ac-e9c05cee7e25)


# Installation
```
git clone https://github.com/mattyamonaca/auto_undercoat.git
cd starline
conda create -n auto_undercoat python=3.10
conda activate auto_undercoat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Usage
1. Image Select.
Select and upload the image you want to undercoat.
下塗りしたい線画画像(背景は透過されている必要があります)を選択してアップロード。

![image](https://github.com/wasanbonplan/auto_undercoat_proto/assets/48423148/9911d330-a561-4bfd-aaf8-d01c7e9cc292)


2. Input Text Prompt.
Enter the color or image you wish to specify and the words that describe it.
指定したい塗の色や、入力した画像を説明する単語・文章をプロンプトとして入力してください。

![スクリーンショット 2023-06-18 083152](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/49ded23d-ab47-46b0-a056-5179fa5841ba)

5. Click 「Start」button.
スタートボタンをクリックしてください


