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
  id 1484
  arrival_time 32239.58656470534
  lifetime 2211.268145646877
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 38
    gpu 40
    rom 47
  ]
  node [
    id 1
    label "1"
    cpu 12
    gpu 1
    rom 49
  ]
  node [
    id 2
    label "2"
    cpu 18
    gpu 16
    rom 14
  ]
  edge [
    source 0
    target 1
    bw 25
  ]
  edge [
    source 1
    target 2
    bw 9
  ]
]
