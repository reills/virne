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
  id 845
  arrival_time 17544.635724300213
  lifetime 3148.8993134623584
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 48
    gpu 13
    rom 13
  ]
  node [
    id 1
    label "1"
    cpu 23
    gpu 4
    rom 20
  ]
  edge [
    source 0
    target 1
    bw 44
  ]
]
