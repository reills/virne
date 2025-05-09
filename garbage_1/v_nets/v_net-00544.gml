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
  id 544
  arrival_time 10282.735627491724
  lifetime 34.06543617182853
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 38
    gpu 27
    rom 41
  ]
  node [
    id 1
    label "1"
    cpu 50
    gpu 47
    rom 33
  ]
  node [
    id 2
    label "2"
    cpu 29
    gpu 24
    rom 21
  ]
  node [
    id 3
    label "3"
    cpu 20
    gpu 39
    rom 39
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 44
    rom 50
  ]
  node [
    id 5
    label "5"
    cpu 0
    gpu 5
    rom 3
  ]
  node [
    id 6
    label "6"
    cpu 24
    gpu 32
    rom 42
  ]
  edge [
    source 0
    target 1
    bw 28
  ]
  edge [
    source 1
    target 2
    bw 50
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 43
  ]
  edge [
    source 4
    target 5
    bw 31
  ]
  edge [
    source 5
    target 6
    bw 35
  ]
]
