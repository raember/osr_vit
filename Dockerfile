ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

COPY environment.yml environment.yml
RUN conda env update -n base --file environment.yml

WORKDIR /osr-vit

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt update
RUN apt install curl git tmux vim zsh -y
RUN echo | RUNZSH=no sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN sed 's/robbyrussell/crcandy/g' -i ~/.zshrc
RUN chsh -s /usr/bin/zsh
RUN pip install wandb
