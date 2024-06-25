# Installation
## Step 1: Clone the Repository
```bash
git clone our_link
```
After cloning, set up the project:
```bash
cd BlinkCode
bash setup_env.sh
```
### Step 2: Install Evaluation Tools
To convert HTML to images, imgkit requires wkhtmltopdf. Follow these steps to download and install wkhtmltopdf:
```bash
mkdir -p ~/wkhtmltopdf
cd ~/wkhtmltopdf
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-3/wkhtmltox-0.12.6-3.archlinux-x86_64.pkg.tar.xz
tar -xf wkhtmltox-0.12.6-3.archlinux-x86_64.pkg.tar.xz
export PATH=~/wkhtmltopdf/usr/local/bin:$PATH
source ~/.bashrc
```
To convert TeX files to images, you need to install TeXLive. You can refer to the following TeXLive installation guide.
- [Installation Guide](./Installl_TeXlive.md)

### step 3: Set OpenAI Key
1. Obtain your OpenAI API key. This can be done by signing up for an account [e.g. here](https://platform.openai.com/), and then creating a key in [account/api-keys](https://platform.openai.com/account/api-keys). 
2. Add the following line to your shell configuration file (e.g., .bashrc or .zshrc):
```bash
export openai_key='your_openai_api_key_here'
```
3. Source the configuration file to apply the changes:
```bash
source ~/.bashrc  # or source ~/.zshrc
```
### Step 4: (Optional)Download Required Models for VIPER
```bash 
cd BlinkCode
bash download_viper_models.sh
cd viper/GLIP
python setup.py clean --all build develop --user
cd ../..
```