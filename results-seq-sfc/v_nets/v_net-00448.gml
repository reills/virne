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
  id 448
  arrival_time 8585.492970634274
  lifetime 1454.1073003972413
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 29
    gpu 50
    rom 49
  ]
  node [
    id 1
    label "1"
    cpu 3
    gpu 10
    rom 16
  ]
  edge [
    source 0
    target 1
    bw 44
  ]
]
