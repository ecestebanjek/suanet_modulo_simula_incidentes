# suanet_modulo_simula_incidentes
Este es el codigo SDM de SUANET para el modulo de simulación de incidentes

# Pasos para crear la imagen de docker
    docker build -t jcastrosdm1/img_mod2:1.0 .
    docker login
    docker push jcastrosdm1/img_mod2:1.0

# Pasos para desplegar en servidor
    docker pull jcastrosdm1/img_mod2:1.0
    docker create -p8502:8501 --name cont_mod2 jcastrosdm1/img_mod2:1.0
    docker start cont_mod2 -d

Nota: El puerto dentro del contenedor siempre sera 8501 para streamlit, y el puerto del servidor (el primero) puede incrementar o variar.
