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
  id 1128
  arrival_time 23506.63732549372
  lifetime 2111.9999162709996
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 34
    gpu 33
    rom 17
  ]
  node [
    id 1
    label "1"
    cpu 41
    gpu 19
    rom 26
  ]
  node [
    id 2
    label "2"
    cpu 22
    gpu 35
    rom 37
  ]
  node [
    id 3
    label "3"
    cpu 50
    gpu 3
    rom 42
  ]
  node [
    id 4
    label "4"
    cpu 37
    gpu 11
    rom 10
  ]
  node [
    id 5
    label "5"
    cpu 39
    gpu 20
    rom 18
  ]
  node [
    id 6
    label "6"
    cpu 47
    gpu 14
    rom 37
  ]
  edge [
    source 0
    target 1
    bw 14
  ]
  edge [
    source 1
    target 2
    bw 0
  ]
  edge [
    source 2
    target 3
    bw 31
  ]
  edge [
    source 3
    target 4
    bw 48
  ]
  edge [
    source 4
    target 5
    bw 24
  ]
  edge [
    source 5
    target 6
    bw 4
  ]
]
