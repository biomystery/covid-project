name=$1
push_aws=false
dir_prefix="dockers/${name}/"
[[ -n $2 ]] && push_aws=true  
version=$(cat ${dir_prefix}/version.txt)
echo "Making $name docker image with $version, push: ${push_aws}"

# build 
tag=389397373118.dkr.ecr.us-west-1.amazonaws.com/${name}:${version}

docker build \
  -t ${tag} \
  -f ${dir_prefix}/Dockerfile \
  .

# login & push 
[[ ${push_aws} = true ]] && aws ecr get-login-password --region us-west-1 | \
  docker login --username AWS --password-stdin 389397373118.dkr.ecr.us-west-1.amazonaws.com

[[ ${push_aws} = true ]] && docker push ${tag}

