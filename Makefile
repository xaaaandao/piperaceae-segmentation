run:
	rm -rf main.py
	jupyter nbconvert main.ipynb --to python
	nohup ipython main.py &

convert:
	jupyter nbconvert main.ipynb --to python
	nohup ipython main.py &

show:
	tail -f nohup.out
