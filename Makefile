build:
	uvx marimo export html-wasm example.py --output docs --mode edit --sandbox --force
	git add .
	git commit -m "main"
	git push origin main