#!/bin/sh

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"cfarr311y@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
