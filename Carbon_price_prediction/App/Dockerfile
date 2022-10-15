FROM mambaorg/micromamba:0.15.3
USER root
RUN mkdir /opt/carbon_price_dashboard
RUN chmod -R 777 /opt/carbon_price_dashboard
WORKDIR /opt/carbon_price_dashboard
USER micromamba
COPY . .
RUN micromamba install -y -n base -f app_env.yml && \
   micromamba clean --all --yes
USER root
RUN chmod a+x run_dashboard.sh
CMD ["./run_dashboard.sh"]
