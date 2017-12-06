all: traces_%.npy

numpy_arrays/%.npy powerblade_arrays/%.npy PLAID/CSV/%.csv:
	./plaid_serializer.py

plaid_paper_batch_data/%.npy: numpy_arrays/%.npy
	./plaid_batching.py

traces_%.npy: plaid_paper_batch_data/%.npy
	./combine_batch_traces.py

clean:
	rm -rf PLAID numpy_arrays powerblade_arrays traces_*.npy
