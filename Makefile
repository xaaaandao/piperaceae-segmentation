run:
	rm -rf main.py
	jupyter nbconvert main.ipynb --to python
	nohup ipython main.py &
	tail -f nohup.out