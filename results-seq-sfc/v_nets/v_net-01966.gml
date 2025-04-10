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
  id 1966
  arrival_time 42969.969307615196
  lifetime 632.222110836286
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 25
    gpu 9
    rom 9
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 11
    rom 47
  ]
  node [
    id 2
    label "2"
    cpu 20
    gpu 20
    rom 26
  ]
  node [
    id 3
    label "3"
    cpu 34
    gpu 10
    rom 5
  ]
  node [
    id 4
    label "4"
    cpu 40
    gpu 0
    rom 12
  ]
  node [
    id 5
    label "5"
    cpu 45
    gpu 10
    rom 39
  ]
  node [
    id 6
    label "6"
    cpu 26
    gpu 23
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 6
  ]
  edge [
    source 1
    target 2
    bw 0
  ]
  edge [
    source 2
    target 3
    bw 47
  ]
  edge [
    source 3
    target 4
    bw 36
  ]
  edge [
    source 4
    target 5
    bw 23
  ]
  edge [
    source 5
    target 6
    bw 33
  ]
]
