check_deps:
	g++ --version
	nvcc --version

clean:
	# build
	rm -rf qptools.egg-info
	rm -rf dist
	rm -rf build
	rm -rf __pycache__
	# test
	rm -f c/core/test/*.o
	rm -f c/core/test/test
	rm -f c/cudacore/test/*.o
	rm -f c/cudacore/test/test