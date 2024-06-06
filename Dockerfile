# Base image
FROM runpod/base:0.4.0-cuda11.8.0

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH=/miniconda/bin:${PATH}

# Create a new conda environment
RUN conda create -n mados python=3.8.12 -y

# Install dependencies in the new environment
RUN /miniconda/bin/conda run -n mados conda install -c conda-forge gdal==3.3.2 -y
COPY builder/requirements.txt /requirements.txt
RUN /miniconda/bin/conda run -n mados pip install --upgrade pip && \
    /miniconda/bin/conda run -n mados pip install --upgrade -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html --no-cache-dir && \
    rm /requirements.txt
RUN /miniconda/bin/conda run -n mados conda install pytables==3.7.0 -y

# Add source files (Worker Template)
ADD marinext .
ADD logs /logs
ADD data /data

# Use ENTRYPOINT to ensure the Conda environment is activated
ENTRYPOINT ["/miniconda/bin/conda", "run", "--no-capture-output", "-n", "mados"]

# Default command
CMD ["bash"]
