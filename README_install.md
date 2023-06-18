# Requirements
Install the requirements.

    pip3 install -r requirements.txt

# Packaging

Download the source code

    git clone https://github.com/trucomanx/PcKor

The next command generates the `dist/PcKor-VERSION.tar.gz` file.

    cd PcKor/src
    python setup.py sdist

For more informations use `python setup.py --help-commands`

# Install 

Install the packaged library

    pip3 install dist/PcKor-VERSION.tar.gz
