FROM nvcr.io/nvidia/deepstream:5.1-21.02-triton as Builder
WORKDIR /opt/ds-pose/
COPY . .
RUN cd deepstream-app && make clean && make

FROM nvcr.io/nvidia/deepstream:5.1-21.02-base
WORKDIR /opt/ds-pose/
COPY --from=Builder /opt/ds-pose/ .
WORKDIR /opt/ds-pose/deepstream-app
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
