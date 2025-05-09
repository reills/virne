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
  id 1864
  arrival_time 40942.99454027451
  lifetime 1297.7279587880207
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 13
    gpu 18
    rom 20
  ]
  node [
    id 1
    label "1"
    cpu 5
    gpu 21
    rom 17
  ]
  node [
    id 2
    label "2"
    cpu 11
    gpu 27
    rom 49
  ]
  node [
    id 3
    label "3"
    cpu 39
    gpu 23
    rom 35
  ]
  node [
    id 4
    label "4"
    cpu 24
    gpu 30
    rom 46
  ]
  edge [
    source 0
    target 1
    bw 14
  ]
  edge [
    source 1
    target 2
    bw 22
  ]
  edge [
    source 2
    target 3
    bw 6
  ]
  edge [
    source 3
    target 4
    bw 25
  ]
]
