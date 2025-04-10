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
  id 335
  arrival_time 6349.887419620482
  lifetime 122.37838140308891
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 41
    gpu 23
    rom 43
  ]
  node [
    id 1
    label "1"
    cpu 20
    gpu 30
    rom 47
  ]
  node [
    id 2
    label "2"
    cpu 16
    gpu 16
    rom 9
  ]
  node [
    id 3
    label "3"
    cpu 16
    gpu 24
    rom 5
  ]
  edge [
    source 0
    target 1
    bw 6
  ]
  edge [
    source 1
    target 2
    bw 23
  ]
  edge [
    source 2
    target 3
    bw 21
  ]
]
