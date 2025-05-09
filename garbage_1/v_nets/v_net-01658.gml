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
  id 1658
  arrival_time 37053.63435068637
  lifetime 1411.3260707938057
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 24
    gpu 24
    rom 9
  ]
  node [
    id 1
    label "1"
    cpu 22
    gpu 34
    rom 18
  ]
  node [
    id 2
    label "2"
    cpu 0
    gpu 41
    rom 25
  ]
  edge [
    source 0
    target 1
    bw 3
  ]
  edge [
    source 1
    target 2
    bw 39
  ]
]
