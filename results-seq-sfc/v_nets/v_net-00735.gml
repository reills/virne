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
  id 735
  arrival_time 15437.031329065358
  lifetime 87.16323953081387
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 10
    gpu 29
    rom 22
  ]
  node [
    id 1
    label "1"
    cpu 27
    gpu 32
    rom 32
  ]
  edge [
    source 0
    target 1
    bw 9
  ]
]
