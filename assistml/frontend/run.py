from assistml_dashboard import create_app

app = create_app()

if __name__ == "__main__":
    app = create_app()
    app.run(host=app.server.config["HOST"], port=app.server.config["PORT"], debug=app.server.config["DEBUG"])
