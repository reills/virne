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
  id 1350
  arrival_time 28606.63016496539
  lifetime 253.59168730267862
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 8
    gpu 5
    rom 1
  ]
  node [
    id 1
    label "1"
    cpu 35
    gpu 14
    rom 0
  ]
  node [
    id 2
    label "2"
    cpu 36
    gpu 41
    rom 44
  ]
  edge [
    source 0
    target 1
    bw 46
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
]
