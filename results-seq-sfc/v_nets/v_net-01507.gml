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
  id 1507
  arrival_time 33585.46823375392
  lifetime 200.66486273810648
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 48
    gpu 13
    rom 31
  ]
  node [
    id 1
    label "1"
    cpu 8
    gpu 46
    rom 46
  ]
  node [
    id 2
    label "2"
    cpu 37
    gpu 22
    rom 12
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 40
  ]
]
