# auto_undercoat
"auto_undercoat" is an automatic undercoat and layer splitter

![スクリーンショット 2023-06-18 083200](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/b87f9a90-ca81-4947-a558-9bc7fac5071c)

![スクリーンショット 2023-06-18 083356](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/f9745298-e428-4ff7-a3ac-e9c05cee7e25)

# Install
1. Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattyamonaca/auto_undercoat/blob/main/launch_app.ipynb).
   
2. Select GPU Runtime.
 ![スクリーンショット 2023-06-18 090105](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/461b6208-3af6-43f4-a7de-2291cf83f5ad)
![スクリーンショット 2023-06-18 085831](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/4f6b6d62-6e63-4e25-8486-7ce604ab17b8)

3. Select 「Run All Cells」.
   
![スクリーンショット 2023-06-18 090400](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/27187080-dd32-4acd-8a38-98085aa36704)

4. Click Ruuning on public URL.

 ![スクリーンショット 2023-06-18 090639](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/11089876-40bf-4308-ac08-f9ff0227d9ea)
 
# Local Install
## Windows Installation
### Required Dependencies
3.10.X > Python > 3.10.8 and Git

### install Step
1. 
```
git clone https://github.com/mattyamonaca/auto_undercoat
```

2. run `install.ps1` first time use, waiting for installation to complete.
3. run `run_gui.ps1` to open local gui.
4. open website localhost:port to use(The default is localhost:7860). 

# Usage
1. Image Select.
Select and upload the image you want to undercoat.
下塗りしたい画像を選択してアップロード。

![スクリーンショット 2023-06-18 082559](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/f72c85ce-902c-4264-870a-09830ad270e9)


2. Select bg_type.
Select "alpha" for a transparent background or "white" for a white background.
背景が透明なら「alpha」を、白色なら「white」を選択してください。

![スクリーンショット 2023-06-18 082758](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/55afc169-805f-42cb-9c01-a1feb2a4865b)


3. Input Text Prompt.
Enter the color or image you wish to specify and the words that describe it.
指定したい塗の色や、入力した画像を説明する単語・文章をプロンプトとして入力してください。
（何も指定しなくても大丈夫です）

![スクリーンショット 2023-06-18 083152](https://github.com/mattyamonaca/auto_undercoat/assets/48423148/49ded23d-ab47-46b0-a056-5179fa5841ba)

5. Click 「Create PSD」button.


