graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1703
  arrival_time 37725.461402282934
  lifetime 715.1978034045671
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 41
    gpu 15
    rom 0
  ]
  node [
    id 1
    label "1"
    cpu 42
    gpu 15
    rom 23
  ]
  node [
    id 2
    label "2"
    cpu 34
    gpu 13
    rom 2
  ]
  edge [
    source 0
    target 1
    bw 41
  ]
  edge [
    source 1
    target 2
    bw 23
  ]
]
