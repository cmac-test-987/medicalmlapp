from streamlit import server

def app(environ, start_response):
    server.server.set_environment(environ, start_response)
    return server.server(environ, start_response)
