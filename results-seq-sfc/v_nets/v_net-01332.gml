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
  id 1332
  arrival_time 28079.72571638474
  lifetime 179.56580285113824
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 20
    gpu 11
    rom 36
  ]
  node [
    id 1
    label "1"
    cpu 19
    gpu 11
    rom 21
  ]
  edge [
    source 0
    target 1
    bw 31
  ]
]
