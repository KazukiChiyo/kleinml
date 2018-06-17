#!/bin/bash
rm -rf *.html
jupyter nbconvert --to html --template basic *.ipynb
